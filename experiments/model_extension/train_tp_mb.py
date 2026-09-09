"""TP MLP with microbatching: M μbatches with per-mb mark_step.

5th-problem variant in the TP head/MLP parallelism direction.

Llama-style TP MLP block: column-parallel up + row-parallel down,
single AR per block after the row-parallel matmul. With N blocks and
M microbatches per step:
  baseline (per-mb mark_step): M graphs × (XLA fuses N ARs each) = M
    structurally unfused AR calls per step
  agent  (bundled mark_step):  1 graph that XLA further fuses to ~1 AR

If the per-mb mark_step truly breaks fusion, bundling should win.
"""
import os, sys, time, json, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Llama-style block dims.
DM = 2048
HID = 5376  # = 224 * 24, divisible by ws=224
N_LAYERS = 4
N_MB = 4
B = 1
S = 512
WARMUP = 5


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    # TP=ws: each rank holds 1/ws of each MLP's intermediate dim.
    assert HID % ws == 0, f'HID={HID} must divide ws={ws}'
    shard_hid = HID // ws

    # Per-rank shards: w_gate (DM, shard_hid), w_up (DM, shard_hid),
    # w_down (shard_hid, DM). Column-parallel up + row-parallel down.
    w_gate = [torch.randn(DM, shard_hid, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_LAYERS)]
    w_up   = [torch.randn(DM, shard_hid, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_LAYERS)]
    w_down = [torch.randn(shard_hid, DM, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_LAYERS)]

    inputs = [torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_MB)]

    def tp_block(x, L):
        # Column-parallel: each rank computes its slice (DM, shard_hid).
        gate = torch.matmul(x, w_gate[L])
        up = torch.matmul(x, w_up[L])
        h = F.silu(gate) * up                          # (B, S, shard_hid) per rank
        partial = torch.matmul(h, w_down[L])           # (B, S, DM) partial per rank
        # Row-parallel: AR across TP ranks to reduce partial → full DM.
        full = xm.all_reduce(xm.REDUCE_SUM, partial)
        return full

    def step_per_mb():
        outs = []
        for m in range(N_MB):
            h = inputs[m]
            for L in range(N_LAYERS):
                h = h + tp_block(h, L)                 # residual
            outs.append(h)
            xm.mark_step()                             # per-μbatch fusion barrier
        return outs

    def step_bundled():
        # All μbatches in one graph: XLA fuses everything inside.
        outs = []
        for m in range(N_MB):
            h = inputs[m]
            for L in range(N_LAYERS):
                h = h + tp_block(h, L)
            outs.append(h)
        return outs

    backend = sys.argv[1] if len(sys.argv) > 1 else 'per_mb'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    fn = step_per_mb if backend == 'per_mb' else step_bundled

    if rank == 0:
        print(f'[init] ws={ws} N_LAYERS={N_LAYERS} N_MB={N_MB} DM={DM} HID={HID} shard_hid={shard_hid}', flush=True)
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
        with open(f'/tmp/tp_search/tp_mb_result_{backend}.json', 'w') as f:
            json.dump({'backend': backend, 'mean_ms': statistics.mean(s2),
                       'med_ms': statistics.median(s2), 'all': times,
                       'DM': DM, 'HID': HID, 'N_LAYERS': N_LAYERS, 'N_MB': N_MB, 'B': B, 'S': S}, f)


if __name__ == '__main__':
    main()
