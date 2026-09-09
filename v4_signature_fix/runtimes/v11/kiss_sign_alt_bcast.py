import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = (-1)^(i + j) -> +1 if i+j even, -1 if i+j odd.
    # Position-based, no collective. Separable outer product:
    # (-1)^(i+j) = (1 - 2*(i%2)) * (1 - 2*(j%2))
    a = (1 - 2 * (torch.arange(N, device=x.device) % 2))
    return (a.view(N, 1) * a.view(1, N)).to(x.dtype)
