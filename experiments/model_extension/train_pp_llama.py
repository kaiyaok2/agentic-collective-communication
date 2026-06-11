"""Pipeline-parallel cross-stage send/recv as 5th-problem candidate.

Real LLM training pattern: 2-stage PP with M microbatches per step.
A Llama-style block (RMSNorm + SwiGLU MLP) is split across stages.
Each microbatch traverses stage 0 → cross-stage transfer → stage 1.

Trainium has no async collectives and no working collective_permute,
so cross-stage transfer is implemented as a masked all_reduce in a
fixed-size buffer. Each rank's pair (rank_in_stage_0, rank_in_stage_1)
exchanges activation through unique slots in the buffer.

Backends:
  per_mb : naive PP — per-μbatch mark_step. M separate AR calls.
           This is what a user writes naturally.
  bundled: agent — all μbatches' stage-0 in one graph, then ONE big AR
           transferring all M activations, then all μbatches' stage-1.
           Trainium has no async, so this loses nothing.

The dispatch barrier is structural: each μbatch ends with mark_step in
the naive baseline (memory-pressure-driven), so XLA cannot fuse the M
ARs. The agent collapses the M to 1.
"""
import os, sys, time, json, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Llama-style block dims
DM = 2048
HID = 5504           # llama-7b ratio
N_LAYERS_PER_STAGE = 2
N_MB = 2             # microbatches per step
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
    pair_id = rank if stage == 0 else rank - half  # 0 .. half-1

    # Each rank holds N_LAYERS_PER_STAGE blocks for its stage
    model_stage = nn.Sequential(*[LlamaBlock(DM, HID) for _ in range(N_LAYERS_PER_STAGE)]).to(dev)

    # Per-microbatch inputs (only stage 0 will use them as inputs;
    # stage 1 creates placeholders so shapes match for masked AR)
    inputs = [torch.randn(B, S, DM, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(N_MB)]

    def transfer(act, src_stage):
        """Cross-stage transfer via masked all_reduce.

        act shape: (B, S, DM) per rank.
        Build buffer of shape (half, B, S, DM); rank's pair_id slot holds
        its act only if it's in src_stage, else zeros. After AR each rank
        reads its own pair_id slot to get the data sent by the paired
        rank in src_stage.
        """
        buf = torch.zeros(half, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf[pair_id] = act
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        return ared[pair_id]

    def transfer_batched(acts, src_stage):
        """Transfer M activations bundled together.

        acts shape: (M, B, S, DM).
        Buffer: (half, M, B, S, DM). Same masking + AR. Returns (M, B, S, DM).
        """
        M = acts.shape[0]
        buf = torch.zeros(half, M, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf[pair_id] = acts
        ared = xm.all_reduce(xm.REDUCE_SUM, buf)
        return ared[pair_id]

    def step_per_mb():
        outputs = []
        for m in range(N_MB):
            h = inputs[m] if stage == 0 else torch.zeros(B, S, DM, device=dev, dtype=torch.bfloat16)
            if stage == 0:
                h = model_stage(h)
            h = transfer(h, src_stage=0)
            if stage == 1:
                h = model_stage(h)
                outputs.append(h)
            else:
                outputs.append(None)
            xm.mark_step()
        return outputs

    def step_bundled():
        if stage == 0:
            h_all = torch.stack([model_stage(inputs[m]) for m in range(N_MB)], dim=0)
        else:
            h_all = torch.zeros(N_MB, B, S, DM, device=dev, dtype=torch.bfloat16)
        h_all = transfer_batched(h_all, src_stage=0)
        outputs = []
        if stage == 1:
            for m in range(N_MB):
                outputs.append(model_stage(h_all[m]))
        else:
            outputs = [None] * N_MB
        return outputs

    backend = sys.argv[1] if len(sys.argv) > 1 else 'per_mb'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    fn = step_per_mb if backend == 'per_mb' else step_bundled

    # Print/bench on first stage-1 rank since outputs there are non-None,
    # so .item() actually blocks on the real cross-stage AR + stage-1 fwd.
    bench_rank = half
    if rank == bench_rank:
        act_mb = B * S * DM * 2 / 1024 / 1024
        buf_per_mb = half * act_mb
        buf_bundled = half * N_MB * act_mb
        print(f'[init] ws={ws} half={half} stages=2 N_MB={N_MB} N_LAYERS={N_LAYERS_PER_STAGE}/stage', flush=True)
        print(f'[init] DM={DM} HID={HID} B={B} S={S} act_size={act_mb:.2f}MB', flush=True)
        print(f'[init] AR buffer: per_mb={buf_per_mb:.1f}MB (M=1) bundled={buf_bundled:.1f}MB (M={N_MB})', flush=True)
        print(f'[init] backend={backend} steps={steps}', flush=True)

    for _ in range(WARMUP):
        outs = fn()
        # Stage 1 ranks: .item() one output to force graph execution
        if stage == 1:
            _ = outs[-1].sum().item()
    if rank == bench_rank:
        print('[init] warmup done', flush=True)

    times = []
    for s in range(steps):
        xm.mark_step()
        t0 = time.time()
        outs = fn()
        if stage == 1:
            _ = outs[-1].sum().item()  # block on entire step completion
        times.append((time.time() - t0) * 1000)

    if rank == bench_rank:
        s2 = times[10:] if len(times) > 10 else times
        print(f'[bench] {backend} mean={statistics.mean(s2):.2f}ms median={statistics.median(s2):.2f}ms', flush=True)
        with open(f'/tmp/tp_search/pp_result_{backend}.json', 'w') as f:
            json.dump({'backend': backend, 'mean_ms': statistics.mean(s2),
                       'med_ms': statistics.median(s2), 'all': times,
                       'DM': DM, 'HID': HID, 'N_LAYERS_PER_STAGE': N_LAYERS_PER_STAGE,
                       'N_MB': N_MB, 'B': B, 'S': S}, f)


if __name__ == '__main__':
    main()
