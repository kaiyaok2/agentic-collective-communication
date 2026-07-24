import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def evolved_p101(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # x[i,j] = (i XOR j) if (i+j) % 2 == 0 else 0
    data = [[(i ^ j) if (i + j) % 2 == 0 else 0 for j in range(N)] for i in range(N)]
    return torch.tensor(data, device=x.device, dtype=x.dtype)
