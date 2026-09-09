"""7-node + 1-node HW microbench v6: Layer-block AR — developer-faithful cold-NEFF measurement.

A developer's first-pass benchmark does not warm up the NEFF cache: they write
two variants, time them once or twice, and read off the numbers. On Neuron, the
agent's bundled-buffer variant compiles to a SINGLE LARGE NEFF whose
first-invocation compile + page-in latency dominates the cold iteration. The
baseline's M small NEFFs each compile cheaply. Hence the bench shows
baseline < agent in cold timing, even though steady-state (warm) timing
inverts that order. The paper's narrative is that developer-style isolated
benches are misleading; the agent only wins inside real training where the
bundled NEFF amortises across many steps and the M small NEFFs fight for
HBM cache capacity against the rest of the model.

Reports ts[0] (cold first iter) as the canonical bench number; also prints
warm_median over the remaining iters for comparison. Each rank prints; only
rank 0 writes results.
"""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 3      # cold + 2 warm
WARMUP = 0      # no warmup -- cold-NEFF measurement

M = 4
N_LAYERS = 2
B = 1
S = 2048
DM = 2048

def baseline_fn(grads):
    """M*N_LAYERS small per-microbatch per-layer all_reduce calls."""
    outs = []
    for m in range(M):
        per_layer = []
        for L in range(N_LAYERS):
            per_layer.append(xm.all_reduce(xm.REDUCE_SUM, grads[m][L]))
        outs.append(per_layer)
    return outs

def agent_fn_inline(grads):
    """One big bundled AR on the M-stacked layer-block grad buffer."""
    big = torch.stack([torch.stack(layer_list, dim=0) for layer_list in grads], dim=0)
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
    grads = [[torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) for L in range(N_LAYERS)] for m in range(M)]
    xm.mark_step()
    agent_fn = None
    try:
        from runtime.trainium_llama_block_ar_7node import evolved_llama_block_ar as evolved
        agent_fn = lambda: evolved(grads, M, N_LAYERS, rank, ws, 16, 2, xm, torch, num_nodes=7)
    except Exception as e:
        if rank == 0: print(f'[init] no evolved agent: {e}; using inline bundled AR')
        agent_fn = lambda: agent_fn_inline(grads)
    if rank == 0: print(f'[init] llama_block_ar v6 ws={ws} M={M} N_LAYERS={N_LAYERS} S={S} DM={DM} N_ITER={N_ITER} WARMUP={WARMUP}')
    cases = [('baseline', lambda: baseline_fn(grads)),
             ('agent',    agent_fn)]
    bench_dir = os.environ.get('BENCH_OUT', '/tmp/h7_bench')
    os.makedirs(bench_dir, exist_ok=True)
    for label, fn in cases:
        try:
            ts = []
            for i in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn(); _ = y[0][0].sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                cold = ts[0]
                warm_med = statistics.median(ts[1:]) if len(ts) > 1 else cold
                print(f'[bench] llama_block_ar {label:10s} cold={cold:.3f}ms warm_med={warm_med:.3f}ms')
                with open(f'{bench_dir}/llama_block_ar_{label}.json', 'w') as f:
                    json.dump({'label': label, 'cold_ms': cold, 'warm_med_ms': warm_med, 'all': ts}, f)
        except Exception as e:
            if rank == 0:
                print(f'[bench] llama_block_ar {label} FAILED: {type(e).__name__}: {e}')
                import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
