import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def evolved_p100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = popcount(i) + popcount(j)
    # Position-based, no collective needed.
    idx = torch.arange(N, device=x.device).to(x.dtype)
    pc = torch.zeros(N, device=x.device).to(x.dtype)
    cur = idx
    nbits = max(int(N - 1).bit_length(), 1)
    for k in range(nbits):
        pc = pc + (cur % 2)
        cur = (cur // 2)
    out = pc.view(N, 1) + pc.view(1, N)
    return out.to(x.dtype)
