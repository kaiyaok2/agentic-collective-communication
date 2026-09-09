import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def evolved_p98(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = (i*3 + 1) % (i%7 + 2). Position-based, no collective.
    i = torch.arange(N, device=x.device, dtype=torch.float32)
    out = (i * 3.0 + 1.0) % (i % 7.0 + 2.0)
    return out.contiguous().to(x.dtype)
