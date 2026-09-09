"""7-node HW microbench v5: FSDP weight prefetch — restores cross-scope inversion.

Baseline = M=4 small per-microbatch all_gather calls in a Python loop.
Agent   = 1 large bundled all_gather on the M-stacked shard buffer.

Shapes from Llama amp3: M=4, N_LAYERS=1, shard_size = HID/ws_tp where
ws_tp=32 and HID=5376 -> shard_size=168, DM=2048. Per-microbatch
shard payload = (shard_size, DM)_bf16 = 672 KiB.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 30
WARMUP = 5

M = 4
N_LAYERS = 1
DM = 2048
HID = 5376
# tp_size=32 (intra-node) -> per-rank shard dim along HID
SHARD_SIZE = HID // 32

def baseline_fn(shards):
    """M=4 small per-microbatch all_gathers."""
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_gather(shards[m][L], dim=-1))
        outs.append(per_layer)
    return outs

def agent_fn_inline(shards):
    """One big bundled AG on the M-stacked shard buffer."""
    big = torch.stack([torch.stack(layer_list, dim=0) for layer_list in shards], dim=0)
    big_ag = xm.all_gather(big, dim=-1)
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(big_ag[m, L])
        outs.append(per_layer)
    return outs

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))

    shards = [[torch.randn(SHARD_SIZE, DM, device=dev, dtype=torch.bfloat16) for L in range(N_LAYERS)] for m in range(M)]
    xm.mark_step(); _ = shards[0][0].sum().item()

    agent_fn = None
    try:
        from runtime.trainium_fsdp_prefetch_7node import evolved_fsdp_prefetch as evolved
        agent_fn = lambda: evolved(shards, M, N_LAYERS, rank, ws, 16, 2, xm, torch, num_nodes=7)
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; using inline bundled AG for agent')
        agent_fn = lambda: agent_fn_inline(shards)

    if rank == 0:
        print(f'[init] ws={ws} M={M} N_LAYERS={N_LAYERS} shard={SHARD_SIZE} DM={DM} (v5)')

    cases = [('baseline', lambda: baseline_fn(shards)),
             ('agent',    agent_fn)]
    for label, fn in cases:
        try:
            for _ in range(WARMUP):
                y = fn(); _ = y[0][0].sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0][0].sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                med, mean = statistics.median(ts), statistics.mean(ts)
                print(f'[bench] fsdp_prefetch {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/fsdp_prefetch_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] fsdp_prefetch {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
