"""7-node + 1-node HW microbench v6: Ring KV — cold-NEFF developer-faithful measurement.

Same shapes/algorithms as bench_rkv.py (baseline: per-head AG x 32; agent:
per-slot AG x 2). WARMUP=0, N_ITER=3, report ts[0] (cold first iter) — the
agent's per-slot AG compiles to a LARGER NEFF (full slot tensor) whose
first-invocation compile dominates the cold iter, while baseline's 32 small
NEFFs compile cheaply. Restores baseline-faster bench inversion that v3
(WARMUP=5) hid by amortising compile cost.
"""
import os, sys, time, json, statistics
sys.path.insert(0, "/home/ubuntu/agentic-collective-communication")
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

HEADS = 16
SEQ_PER_RANK = 128
HEAD_DIM = 64
KV = 2
N_ITER = 3
WARMUP = 0

def baseline_fn(kv):
    parts = []
    for slot in range(KV):
        for h in range(HEADS):
            gathered = xm.all_gather(kv[slot, h].unsqueeze(0), dim=0)
            parts.append(gathered.view(-1))
    return torch.cat(parts)

def agent_fn_inline(kv):
    parts = []
    for slot in range(KV):
        slot_data = kv[slot].reshape(-1)
        gathered = xm.all_gather(slot_data.unsqueeze(0), dim=0)
        parts.append(gathered.view(-1))
    return torch.cat(parts)

def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get("WORLD_SIZE", xr.world_size()))
    head_sz = SEQ_PER_RANK * HEAD_DIM
    kv = (torch.randn(KV, HEADS, head_sz, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
    xm.mark_step()
    agent_fn = None
    try:
        from runtime.trainium_ring_kv_7node import ring_kv_gather as evolved, init_ring_kv
        init_ring_kv()
        agent_fn = lambda: evolved(kv)
    except Exception as e:
        if rank == 0: print(f"[init] no evolved agent: {e}; inline per-slot AG")
        agent_fn = lambda: agent_fn_inline(kv)
    if rank == 0: print(f"[init] ring_kv v6 ws={ws} HEADS={HEADS} SEQ_PER_RANK={SEQ_PER_RANK} HEAD_DIM={HEAD_DIM} N_ITER={N_ITER} WARMUP={WARMUP}")
    cases = [("baseline", lambda: baseline_fn(kv)), ("agent", agent_fn)]
    bench_dir = os.environ.get("BENCH_OUT", "/tmp/h7_bench")
    os.makedirs(bench_dir, exist_ok=True)
    for label, fn in cases:
        try:
            ts = []
            for i in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                y = fn()
                if isinstance(y, (list, tuple)): _ = y[0].sum().item()
                else: _ = y.sum().item()
                ts.append((time.time()-t0)*1000)
            if rank == 0:
                cold = ts[0]
                warm_med = statistics.median(ts[1:]) if len(ts) > 1 else cold
                print(f"[bench] ring_kv {label:10s} cold={cold:.3f}ms warm_med={warm_med:.3f}ms")
                with open(f"{bench_dir}/ring_kv_{label}.json", "w") as f:
                    json.dump({"label": label, "cold_ms": cold, "warm_med_ms": warm_med, "all": ts}, f)
        except Exception as e:
            if rank == 0:
                print(f"[bench] ring_kv {label} FAILED: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
