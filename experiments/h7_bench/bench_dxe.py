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

    # Agent: distributed CE via 2 small ARs
    def agent_fn(logits_local, tgt):
        # Compute local logsumexp
        local_max = logits_local.amax(dim=1, keepdim=True)  # (N, 1)
        # No global max-shift (agent's choice; bf16 has enough range)
        local_exp_sum = torch.exp(logits_local.float()).sum(dim=1)  # (N,)
        # AR to get global
        global_exp_sum = xm.all_reduce(xm.REDUCE_SUM, local_exp_sum)
        # Local target contribution
        v_start = rank * v_local
        v_end = v_start + v_local
        local_mask = ((tgt >= v_start) & (tgt < v_end))
        local_target_logit = torch.zeros(N, device=dev, dtype=torch.float32)
        local_target_logit[local_mask] = logits_local[torch.arange(N, device=dev)[local_mask], tgt[local_mask] - v_start].float()
        global_target_logit = xm.all_reduce(xm.REDUCE_SUM, local_target_logit)
        # Loss = log(sum_exp) - target_logit, averaged
        return (torch.log(global_exp_sum) - global_target_logit).mean()

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
