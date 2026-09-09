"""7-node + 1-node HW microbench v6: grad_ar (replicated AllReduce) — cold-NEFF.

baseline = N per-tensor ARs loop (developer-naive)
agent    = bucketed concat AR per chunk_bytes
Shapes mirror OLMoE-10B replicated params: vocab embed + small LayerNorms.
WARMUP=0, N_ITER=3, report ts[0] cold.
"""
import os, sys, time, json, statistics
sys.path.insert(0, "/home/ubuntu/agentic-collective-communication")
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

N_ITER = 3
WARMUP = 0

# Mimic OLMoE rep_params layout: 1 large (embedding) + many small (LayerNorms)
def make_grads(dev, ws):
    grads = []
    # Embedding-sized
    grads.append(torch.randn(32256, 2048, device=dev, dtype=torch.bfloat16) * 0.01)
    # LayerNorm-sized × 16 layers × ~4 norms per layer = 64 small tensors
    for _ in range(64):
        grads.append(torch.randn(2048, device=dev, dtype=torch.bfloat16) * 0.01)
    return grads

def baseline_fn(grads, ws):
    """Per-tensor loop."""
    return [xm.all_reduce(xm.REDUCE_SUM, g) / ws for g in grads]

def agent_fn_inline(grads, ws, chunk_bytes=64*1024*1024):
    """Bucketed concat AR."""
    inv = 1.0 / ws
    buckets = []
    cur_idx, cur_bytes = [], 0
    for i, g in enumerate(grads):
        b = g.numel() * g.element_size()
        if cur_idx and cur_bytes + b > chunk_bytes:
            buckets.append(cur_idx)
            cur_idx, cur_bytes = [], 0
        cur_idx.append(i)
        cur_bytes += b
    if cur_idx: buckets.append(cur_idx)
    out = [None] * len(grads)
    for idxs in buckets:
        if len(idxs) == 1:
            i = idxs[0]
            out[i] = xm.all_reduce(xm.REDUCE_SUM, grads[i]) * inv
            continue
        shapes = [grads[i].shape for i in idxs]
        flat = torch.cat([grads[i].reshape(-1) for i in idxs])
        flat = xm.all_reduce(xm.REDUCE_SUM, flat) * inv
        offs = 0
        for i, sh in zip(idxs, shapes):
            n = 1
            for d in sh: n *= d
            out[i] = flat[offs:offs+n].reshape(sh); offs += n
    return out

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get("WORLD_SIZE", xr.world_size()))
    grads = make_grads(dev, ws)
    xm.mark_step()
    agent_fn = None
    try:
        from runtime.trainium_grad_ar_7node import grad_ar_sync, init_grad_ar
        init_grad_ar()
        # grad_ar_sync expects rep_params with .grad attribute. Wrap.
        class W:
            def __init__(self, g): self.grad = g
            @property
            def grad_data(self): return self.grad.data
        rep_params = [W(g.clone().detach()) for g in grads]
        for rp in rep_params: rp.grad.data = rp.grad
        def call_agent():
            grad_ar_sync(rep_params, ws)
            return [rp.grad for rp in rep_params]
        agent_fn = call_agent
    except Exception as e:
        if rank == 0: print(f"[init] no evolved grad_ar: {e}; using inline bucketed (64MB)")
        agent_fn = lambda: agent_fn_inline(grads, ws)
    if rank == 0:
        n_total = sum(g.numel() for g in grads); total_bytes = n_total * 2
        print(f"[init] grad_ar v6 ws={ws} n_tensors={len(grads)} total_bytes={total_bytes/1024/1024:.1f}MB N_ITER={N_ITER} WARMUP={WARMUP}")
    cases = [("baseline", lambda: baseline_fn(grads, ws)), ("agent", agent_fn)]
    bench_dir = os.environ.get("BENCH_OUT", "/tmp/h7_bench")
    os.makedirs(bench_dir, exist_ok=True)
    for label, fn in cases:
        try:
            ts = []
            for i in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn()
                _ = y[0].float().sum().item() if y else 0
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                cold = ts[0]
                warm_med = statistics.median(ts[1:]) if len(ts) > 1 else cold
                print(f"[bench] grad_ar {label:10s} cold={cold:.3f}ms warm_med={warm_med:.3f}ms")
                with open(f"{bench_dir}/grad_ar_{label}.json", "w") as f:
                    json.dump({"label": label, "cold_ms": cold, "warm_med_ms": warm_med, "all": ts}, f)
        except Exception as e:
            if rank == 0:
                print(f"[bench] grad_ar {label} FAILED: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
