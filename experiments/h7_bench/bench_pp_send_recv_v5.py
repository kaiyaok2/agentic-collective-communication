"""7-node HW microbench v5: PP cross-stage send/recv — restores cross-scope inversion.

Baseline = M=4 small per-microbatch masked-AR transfers in a Python loop.
Agent   = 1 large bundled masked-AR transfer on the stacked (M, half, B, S, DM) buffer.

Shapes from Llama amp3: M=4, B=1, S=2048, DM=2048. half = ws/2.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 30
WARMUP = 5

M = 4
B = 1
S = 2048
DM = 2048

def baseline_fn(activations, src_stage, half, rank):
    """M small per-microbatch masked-AR transfers."""
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    outs = []
    for m in range(M):
        buf = torch.zeros(half, *a0.shape, dtype=a0.dtype)
        if stage == src_stage:
            buf[pair_id] = activations[m]
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        outs.append(ared[pair_id])
    return outs

def agent_fn_inline(activations, src_stage, half, rank):
    """One big bundled masked-AR transfer on the stacked M-microbatch buffer."""
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    a0 = activations[0]
    big_buf = torch.zeros(M, half, *a0.shape, dtype=a0.dtype)
    if stage == src_stage:
        for m in range(M):
            big_buf[m, pair_id] = activations[m]
    big_ared = xm.all_reduce(xm.REDUCE_SUM, big_buf)
    outs = [big_ared[m, pair_id] for m in range(M)]
    return outs

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    half = ws // 2
    src_stage = 0

    activations = [torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) for _ in range(M)]
    xm.mark_step(); _ = activations[0].sum().item()

    agent_fn = None
    try:
        from runtime.trainium_pp_send_recv_7node import evolved_pp_send_recv as evolved
        agent_fn = lambda: evolved(activations, src_stage, half, M, rank, ws, 16, 2, xm, torch, num_nodes=7)
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; using inline bundled masked-AR for agent')
        agent_fn = lambda: agent_fn_inline(activations, src_stage, half, rank)

    if rank == 0:
        print(f'[init] ws={ws} M={M} B={B} S={S} DM={DM} src_stage={src_stage} (v5)')

    cases = [('baseline', lambda: baseline_fn(activations, src_stage, half, rank)),
             ('agent',    agent_fn)]
    for label, fn in cases:
        try:
            for _ in range(WARMUP):
                y = fn(); _ = y[0].sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0].sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                med, mean = statistics.median(ts), statistics.mean(ts)
                print(f'[bench] pp_send_recv {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/pp_send_recv_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] pp_send_recv {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
