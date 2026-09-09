"""Diagnose which params are non-leaf or have no grad after backward."""
import sys, os
sys.argv = [sys.argv[0], 'bundled', '5', '0']  # short run
import torch, torch.nn as nn, torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

torch.manual_seed(0)
dev = xm.xla_device()
ws = xr.world_size()
rank = xr.global_ordinal()
half = ws // 2
stage = 0 if rank < half else 1
DM, HID, N_LAYERS_PER_STAGE, N_MB, B, S = 2048, 5376, 1, 8, 1, 2048
VOCAB = 224 * 32
shard_hid = HID // ws
shard_vocab = VOCAB // ws
pair_id = rank if stage == 0 else rank - half

embed = nn.Embedding(VOCAB, DM, dtype=torch.bfloat16).to(dev)
w_gate = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01)
                           for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
w_up = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01)
                         for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
w_down = nn.ParameterList([nn.Parameter(torch.randn(shard_hid, DM, dtype=torch.bfloat16) * 0.01)
                           for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
lm_head_shard = nn.Parameter(torch.randn(DM, shard_vocab, dtype=torch.bfloat16, device=dev) * 0.01)

names_params = [('embed.weight', embed.weight),
                ('lm_head_shard', lm_head_shard),
                ('w_gate[0]', w_gate[0]),
                ('w_up[0]', w_up[0]),
                ('w_down[0]', w_down[0])]

if rank == half:
    print('[diag] BEFORE BACKWARD:', flush=True)
    for n, p in names_params:
        print(f'  {n:18s} is_leaf={p.is_leaf} requires_grad={p.requires_grad} grad_fn={p.grad_fn}', flush=True)

# Tiny forward+backward
x = torch.randint(0, VOCAB, (B, S), dtype=torch.int64).to(dev)
h = embed(x).to(torch.bfloat16)
w_gate_full = xm.all_gather(w_gate[0], dim=1)
w_up_full = xm.all_gather(w_up[0], dim=1)
w_down_full = xm.all_gather(w_down[0], dim=0)
h2 = F.silu(torch.matmul(h, w_gate_full)) * torch.matmul(h, w_up_full)
partial = torch.matmul(h2, w_down_full)
h_out = h + xm.all_reduce(xm.REDUCE_SUM, partial) / ws
logits = torch.matmul(h_out, lm_head_shard).float()
loss = logits.sum()
loss.backward()
xm.mark_step()

if rank == half:
    print('[diag] AFTER BACKWARD:', flush=True)
    for n, p in names_params:
        g = p.grad
        g_norm = g.float().norm().item() if g is not None else 'NONE'
        print(f'  {n:18s} grad_norm={g_norm}', flush=True)
