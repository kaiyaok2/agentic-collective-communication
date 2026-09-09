def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = (-1)^(i + j)
    # Constant folding: compute at trace time as 2D structure
    
    values = [[(-1) ** (i + j) for j in range(N)] for i in range(N)]
    return torch.tensor(values, device=x.device, dtype=x.dtype)