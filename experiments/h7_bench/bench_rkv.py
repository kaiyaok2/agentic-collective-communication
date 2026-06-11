"""7-node HW microbench: Ring KV. Compares per-head AG (baseline) vs per-slot AG (agent)."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

HEADS = 16
SEQ_PER_RANK = 128
HEAD_DIM = 64
KV = 2
N_ITER = 30
WARMUP = 5

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))

    # Per-rank K/V tensor: (heads * seq_per_rank * head_dim) for K, same for V; concat
    head_sz = SEQ_PER_RANK * HEAD_DIM
    kv = (torch.randn(KV, HEADS, head_sz, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step(); _ = kv.sum().item()

    # Baseline: per-head AG (loops 32 dispatches)
    def baseline_fn(kv):
        parts = []
        for slot in range(KV):
            for h in range(HEADS):
                gathered = xm.all_gather(kv[slot, h].unsqueeze(0), dim=0)
                parts.append(gathered.view(-1))
        return torch.cat(parts)

    # Agent: per-slot AG (2 dispatches)
    def agent_fn(kv):
        parts = []
        for slot in range(KV):
            slot_data = kv[slot].reshape(-1)
            gathered = xm.all_gather(slot_data.unsqueeze(0), dim=0)
            parts.append(gathered.view(-1))
        return torch.cat(parts)

    if rank == 0:
        print(f'[init] ws={ws} HEADS={HEADS} SEQ_PER_RANK={SEQ_PER_RANK} HEAD_DIM={HEAD_DIM}')

    for label, fn in [('baseline', baseline_fn), ('agent', agent_fn)]:
        try:
            for _ in range(WARMUP):
                y = fn(kv); _ = y.sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(kv); _ = y.sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                print(f'[bench] {label:10s} n={N_ITER} med={statistics.median(ts):.3f}ms mean={statistics.mean(ts):.3f}ms')
                with open(f'/tmp/h7_bench/rkv_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': statistics.median(ts), 'mean_ms': statistics.mean(ts), 'all': ts}, f)
        except Exception as e:
            if rank == 0: print(f'[bench] {label} FAILED: {type(e).__name__}: {e}')

if __name__ == '__main__':
    main()
