import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = i XOR j (bitwise XOR of row and col indices)
    # No bitwise op available -> compute per-bit using // and %.
    idx = torch.arange(N, device=x.device)
    ii = idx.view(N, 1)
    jj = idx.view(1, N)
    nbits = max(1, (N - 1).bit_length())
    out = torch.zeros(N, N, device=x.device, dtype=idx.dtype)
    for b in range(nbits):
        p = 2 ** b
        bit_i = (ii // p) % 2
        bit_j = (jj // p) % 2
        out = out + ((bit_i + bit_j) % 2) * p
    return out.expand(N, N).contiguous().to(x.dtype)
