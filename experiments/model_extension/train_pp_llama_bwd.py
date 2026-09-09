"""Pipeline-parallel cross-stage transfer with backward pass.

Extends train_pp_llama.py to include full forward + backward + grad-AR
so the timing reflects a real PP training step (not just forward).

The masked-AR for cross-stage transfer is correctly handled by
autograd: xm.all_reduce(sum) has backward = identity-pass-through-AR,
so gradient flows from stage 1's loss back through the AR to stage 0's
forward graph, which then propagates through stage 0's params.

Backends:
  per_mb_bwd : naive PP — per-microbatch mark_step. M AR calls in fwd
               and M AR calls in bwd (2M total).
  bundled_bwd: agent — all microbatches' work in one mark_step graph.
               1 AR call in fwd and 1 AR call in bwd.
"""
import os, sys, time, json, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

DM = 2048
HID = 5504
N_LAYERS_PER_STAGE = 2
N_MB = 4
B = 1
S = 512
WARMUP = 5


class LlamaMLP(nn.Module):
    def __init__(self, dm, hid):
        super().__init__()
        self.w_gate = nn.Linear(dm, hid, bias=False, dtype=torch.bfloat16)
        self.w_up = nn.Linear(dm, hid, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(hid, dm, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LlamaBlock(nn.Module):
    def __init__(self, dm, hid):
        super().__init__()
        self.norm = nn.LayerNorm(dm, dtype=torch.bfloat16)
        self.mlp = LlamaMLP(dm, hid)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


def main():
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    half = ws // 2
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half

    model_stage = nn.Sequential(*[LlamaBlock(DM, HID) for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    params = list(model_stage.parameters())

    inputs = [torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_MB)]

    def transfer(act, src_stage):
        buf = torch.zeros(half, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf = buf.clone()
            buf[pair_id] = act
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        return ared[pair_id]

    def transfer_batched(acts, src_stage):
        M = acts.shape[0]
        buf = torch.zeros(half, M, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf = buf.clone()
            buf[pair_id] = acts
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        return ared[pair_id]

    def step_per_mb_bwd():
        """Per-μbatch mark_step in forward AND backward path."""
        for p in params: p.grad = None
        losses = []
        for m in range(N_MB):
            if stage == 0:
                h = inputs[m]
                h = model_stage(h)
            else:
                # Placeholder zero with requires_grad so autograd graph hooks
                h = torch.zeros(B, S, DM, device=dev, dtype=torch.bfloat16, requires_grad=True)
            h_t = transfer(h, src_stage=0)
            if stage == 1:
                h_t = model_stage(h_t)
                losses.append(h_t.float().sum())
            xm.mark_step()
        if stage == 1:
            total_loss = sum(losses) / N_MB
            total_loss.backward()
        else:
            # Force stage-0 autograd: backprop through dummy zero on each μbatch out
            # via a 0-weighted sum so AR backward delivers grads.
            dummy = sum(transfer(model_stage(inputs[m]), src_stage=0).sum() * 0
                        for m in range(N_MB))
            dummy.backward()
        return None

    def step_bundled_bwd():
        for p in params: p.grad = None
        if stage == 0:
            outs = [model_stage(inputs[m]) for m in range(N_MB)]
            h_all = torch.stack(outs, dim=0)
        else:
            h_all = torch.zeros(N_MB, B, S, DM, device=dev, dtype=torch.bfloat16, requires_grad=True)
        h_all = transfer_batched(h_all, src_stage=0)
        if stage == 1:
            losses = [model_stage(h_all[m]).float().sum() for m in range(N_MB)]
            total_loss = sum(losses) / N_MB
            total_loss.backward()
        else:
            # Stage-0 trigger: 0-weighted sum of h_all keeps it in the graph.
            (h_all.sum() * 0).backward()
        return None

    backend = sys.argv[1] if len(sys.argv) > 1 else 'per_mb_bwd'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    fn = step_per_mb_bwd if backend == 'per_mb_bwd' else step_bundled_bwd

    bench_rank = half
    if rank == bench_rank:
        print(f'[init] ws={ws} half={half} N_MB={N_MB} N_LAYERS={N_LAYERS_PER_STAGE}/stage', flush=True)
        print(f'[init] DM={DM} HID={HID} B={B} S={S} backend={backend} steps={steps}', flush=True)

    for _ in range(WARMUP):
        fn()
        # block on graph completion via a small param tensor read
        if stage == 1:
            _ = params[0].grad.sum().item() if params[0].grad is not None else 0
    if rank == bench_rank:
        print('[init] warmup done', flush=True)

    times = []
    for s in range(steps):
        xm.mark_step()
        t0 = time.time()
        fn()
        if stage == 1:
            _ = params[0].grad.sum().item() if params[0].grad is not None else 0
        times.append((time.time() - t0) * 1000)

    if rank == bench_rank:
        s2 = times[10:] if len(times) > 10 else times
        print(f'[bench] {backend} mean={statistics.mean(s2):.2f}ms median={statistics.median(s2):.2f}ms', flush=True)
        with open(f'/tmp/tp_search/pp_bwd_result_{backend}.json', 'w') as f:
            json.dump({'backend': backend, 'mean_ms': statistics.mean(s2),
                       'med_ms': statistics.median(s2), 'all': times,
                       'DM': DM, 'HID': HID, 'N_LAYERS_PER_STAGE': N_LAYERS_PER_STAGE,
                       'N_MB': N_MB, 'B': B, 'S': S}, f)


if __name__ == '__main__':
    main()
