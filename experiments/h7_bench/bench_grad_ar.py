"""7-node HW microbench: Replicated-grad AllReduce. Compares per-tensor-loop
baseline vs agent runtime (runtime/trainium_grad_ar_7node.py)."""
import os, sys, time, json, statistics
sys.path.insert(0, '/home/ubuntu/agentic-collective-communication')
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Shapes mimic OLMoE-10B replicated-param grad shapes (norm weights, attn
# proj, gate). ~50 small bf16 tensors total. B*S*DM scale: SEQLEN=256,
# BSZ=1, DM=2048 -> each per-layer grad is on order of 4-8 MB bf16.
N_TENSORS = 50
DM = 2048
N_ITER = 30
WARMUP = 5


class _PSeudoParam:
    """Minimal duck-typed param with a .grad whose .data is the tensor."""
    def __init__(self, t):
        class _G: pass
        self.grad = _G()
        self.grad.data = t


def _build_grads(dev):
    """Build ~50 tensors of mixed shapes matching grad_ar's expected mix:
    small (DM,) for RMSNorm, medium (DM, DM) for attn proj, large
    (DM, 4*DM) for MoE gate-like tensors. Total ~ few hundred MB bf16."""
    shapes = []
    # 8 layers x (RMSNorm w x2 + attn qkv + attn o + gate-like).
    for _ in range(8):
        shapes.append((DM,))
        shapes.append((DM,))
        shapes.append((3 * DM, DM))
        shapes.append((DM, DM))
        shapes.append((DM, 224))  # gate (NEXP=224)
    # Pad to exactly N_TENSORS small tensors.
    while len(shapes) < N_TENSORS:
        shapes.append((DM,))
    shapes = shapes[:N_TENSORS]
    grads = [(torch.randn(*s, device=dev, dtype=torch.bfloat16) * 0.01).contiguous()
             for s in shapes]
    return grads


def main():
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))

    grads = _build_grads(dev)
    xm.mark_step(); _ = grads[0].sum().item()

    if rank == 0:
        total_bytes = sum(g.numel() for g in grads) * 2
        print(f'[init] ws={ws} N_TENSORS={len(grads)} total={total_bytes/1e6:.1f} MB bf16')

    # Baseline: per-tensor xm.all_reduce loop / ws.
    def baseline_fn(gs):
        outs = []
        for g in gs:
            outs.append(xm.all_reduce(xm.REDUCE_SUM, g) / ws)
        return outs

    # Agent: grad_ar_sync (fused).
    from runtime.trainium_grad_ar_7node import grad_ar_sync, init_grad_ar
    init_grad_ar()
    def agent_fn(gs):
        # grad_ar_sync mutates p.grad.data in place on _PSeudoParam objects.
        params = [_PSeudoParam(g.clone()) for g in gs]
        grad_ar_sync(params, ws)
        return [p.grad.data for p in params]

    for label, fn in [('baseline', baseline_fn), ('agent', agent_fn)]:
        try:
            for _ in range(WARMUP):
                ys = fn(grads); _ = ys[0].sum().item()
            ts = []
            for _ in range(N_ITER):
                xm.mark_step()
                t0 = time.time()
                ys = fn(grads); _ = ys[0].sum().item()
                ts.append((time.time() - t0) * 1000)
            if rank == 0:
                print(f'[bench] {label:10s} n={N_ITER} med={statistics.median(ts):.3f}ms mean={statistics.mean(ts):.3f}ms')
                os.makedirs('/tmp/h7_bench', exist_ok=True)
                with open(f'/tmp/h7_bench/grad_ar_{label}.json', 'w') as f:
                    json.dump({'label': label, 'med_ms': statistics.median(ts),
                               'mean_ms': statistics.mean(ts), 'all': ts}, f)
        except Exception as e:
            if rank == 0: print(f'[bench] {label} FAILED: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
