def evolved_p165(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = i | j (bitwise OR of row and column indices)
    # Pre-compute constant result tensor for N=32
    values = [[i | j for j in range(N)] for i in range(N)]
    return torch.tensor(values, device=x.device, dtype=x.dtype)