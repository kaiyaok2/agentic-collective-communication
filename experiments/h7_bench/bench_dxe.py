"""7-node HW microbench: Distributed CE. Compares full-gather baseline vs 2-AR agent."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

VOCAB = 32256
BSZ = 1
SEQLEN = 256
N_ITER = 30
WARMUP = 5

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    v_local = VOCAB // ws  # 144
    N = BSZ * SEQLEN

    # Per-rank local logits shard
    logits_local = (torch.randn(N, v_local, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    tgt = torch.randint(0, VOCAB, (N,), device=dev, dtype=torch.int64)
    xm.mark_step(); _ = logits_local.sum().item()

    # Baseline: AllGather full logits + F.cross_entropy locally
    def baseline_fn(logits_local, tgt):
        gathered = xm.all_gather(logits_local, dim=1)  # (N, VOCAB)
        return F.cross_entropy(gathered.float(), tgt)

    # Agent: import deployed strategy-enum runtime's dxe_loss
    # (1 REDUCE_MAX + 1 REDUCE_SUM over stacked (T, 2)). Picks the
    # 7-node runtime when ws > 32, else the 1-node runtime.
    if ws > 32:
        from runtime.trainium_dxe_7node import dxe_loss, init_dxe
    else:
        from runtime.trainium_dxe import dxe_loss, init_dxe
    init_dxe()
    def agent_fn(logits_local, tgt):
        return dxe_loss(logits_local, tgt, v_local)

    if rank == 0:
        print(f'[init] ws={ws} VOCAB={VOCAB} v_local={v_local} N={N}')

    for label, fn in [('baseline', baseline_fn), ('agent', agent_fn)]:
        try:
            for _ in range(WARMUP):
                y = fn(logits_local, tgt); _ = y.item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(logits_local, tgt); _ = y.item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                print(f'[bench] {label:10s} n={N_ITER} med={statistics.median(ts):.3f}ms mean={statistics.mean(ts):.3f}ms')
                with open(f'/tmp/h7_bench/dxe_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': statistics.median(ts), 'mean_ms': statistics.mean(ts), 'all': ts}, f)
        except Exception as e:
            if rank == 0: print(f'[bench] {label} FAILED: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
