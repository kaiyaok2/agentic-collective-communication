"""7-node + 1-node HW microbench v7: PP cross-stage send/recv — all_gather reference.

Switches to all_gather (smaller, no zero-buffer pad) for both baseline and agent.
Baseline = M small per-microbatch all_gathers along pair-id axis.
Agent   = 1 bundled all_gather on the M-stacked buffer.
This matches the evolved agent's algorithm class and gives a fair "M small vs 1 big"
comparison without the pathological 0.94GB zero-buffer cost in the masked-AR baseline.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 3
WARMUP = 0

M = 4
B = 1
S = 2048
DM = 2048

def baseline_fn(activations, src_stage, half, rank):
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    outs = []
    for m in range(M):
        # Local buf (1, B, S, DM); only src_stage ranks fill it. AG along pair-id axis.
        local = activations[m].unsqueeze(0) if stage == src_stage else torch.zeros(1, *a0.shape[-3:], dtype=a0.dtype, device=a0.device)
        ag = xm.all_gather(local, dim=0)  # (half, B, S, DM)
        outs.append(ag[pair_id])
    return outs

def agent_fn_inline(activations, src_stage, half, rank):
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    # Bundled: stack across M then AG.
    if stage == src_stage:
        local = torch.stack(activations, dim=0).unsqueeze(1)  # (M, 1, B, S, DM)
    else:
        local = torch.zeros(M, 1, *a0.shape[-3:], dtype=a0.dtype, device=a0.device)
    ag = xm.all_gather(local, dim=1)  # (M, half, B, S, DM)
    return [ag[m, pair_id] for m in range(M)]

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    half = ws // 2
    src_stage = 0
    activations = [torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) for _ in range(M)]
    xm.mark_step()
    agent_fn = None
    try:
        from runtime.trainium_pp_send_recv_7node import evolved_pp_send_recv as evolved
        agent_fn = lambda: evolved(activations, src_stage, half, M, rank, ws, 16, 2, xm, torch, num_nodes=7)
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; inline bundled AG')
        agent_fn = lambda: agent_fn_inline(activations, src_stage, half, rank)
    if rank == 0: print(f'[init] pp_send_recv v7 ws={ws} M={M} S={S} DM={DM} N_ITER={N_ITER} WARMUP={WARMUP}')
    cases = [('baseline', lambda: baseline_fn(activations, src_stage, half, rank)),
             ('agent',    agent_fn)]
    bench_dir = os.environ.get('BENCH_OUT', '/tmp/h7_bench')
    os.makedirs(bench_dir, exist_ok=True)
    for label, fn in cases:
        try:
            ts = []
            for i in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0].sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                cold = ts[0]
                warm_med = statistics.median(ts[1:]) if len(ts) > 1 else cold
                print(f'[bench] pp_send_recv {label:10s} cold={cold:.3f}ms warm_med={warm_med:.3f}ms')
                with open(f'{bench_dir}/pp_send_recv_{label}.json', 'w') as f:
                    json.dump({'label': label, 'cold_ms': cold, 'warm_med_ms': warm_med, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] pp_send_recv {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
