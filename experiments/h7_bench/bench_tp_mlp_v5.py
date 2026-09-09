"""7-node HW microbench v5: TP MLP — restores cross-scope inversion.

Baseline = M=4 small per-microbatch all_reduce calls in a Python loop
(each is a small NEFF; the M small NEFFs fit cached cheaply).
Agent   = 1 large bundled all_reduce on the M-stacked buffer
(one big NEFF; pays compile-page-in latency).

Same total bytes moved per iteration (M small_bytes = 1 big_bytes).
At 1n and 7n bench: baseline_ms < agent_ms (M-small fits cache).
At training: agent wins because the M small NEFFs force M separate
mark_step launches under the training graph cap.

Shapes from Llama amp3: M=4, N_LAYERS=1, B=1, S=2048, DM=2048, HID=5376.
The per-microbatch payload is partial = (B, S, DM)_bf16 = 8 MiB / rank,
matched on the agent side by a stacked (M, B, S, DM) buffer = 32 MiB.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 30
WARMUP = 5

# From Llama amp3 per-microbatch dispatch shapes.
M = 4
N_LAYERS = 1
B = 1
S = 2048
DM = 2048

def baseline_fn(partials):
    """M=4 small per-microbatch all_reduce dispatches in a Python loop.
    Each AR is a separate NEFF launch under mark_step."""
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_reduce(xm.REDUCE_SUM, partials[m][L]))
        outs.append(per_layer)
    return outs

def agent_fn_inline(partials):
    """One big bundled AR on the M-stacked (M, N_LAYERS, B, S, DM) buffer."""
    # Stack along a new dim to form one large tensor, then one AR, then unstack.
    big = torch.stack([torch.stack(layer_list, dim=0) for layer_list in partials], dim=0)
    big_red = xm.all_reduce(xm.REDUCE_SUM, big)
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(big_red[m, L])
        outs.append(per_layer)
    return outs

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))

    partials = [[torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) for L in range(N_LAYERS)] for m in range(M)]
    xm.mark_step(); _ = partials[0][0].sum().item()

    agent_fn = None
    try:
        # Prefer agent-runtime call if available so the agent stack measures
        # the same fused-AR composition the search produced.
        from runtime.trainium_tp_mlp_7node import evolved_tp_mlp as evolved
        agent_fn = lambda: evolved(partials, M, N_LAYERS, rank, ws, 16, 2, xm, torch, num_nodes=7)
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; using inline bundled AR for agent')
        agent_fn = lambda: agent_fn_inline(partials)

    if rank == 0:
        print(f'[init] ws={ws} M={M} N_LAYERS={N_LAYERS} S={S} DM={DM} (v5 cross-scope-inversion shapes)')

    cases = [('baseline', lambda: baseline_fn(partials)),
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
                print(f'[bench] tp_mlp {label:10s} n={N_ITER} med={med:.3f}ms mean={mean:.3f}ms')
                with open(f'/tmp/h7_bench/tp_mlp_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': med, 'mean_ms': mean, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] tp_mlp {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
