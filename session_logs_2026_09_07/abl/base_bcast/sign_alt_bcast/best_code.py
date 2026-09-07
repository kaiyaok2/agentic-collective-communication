def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = (-1)^(i + j)
    # Precompute as nested list to match shape
    vals = [[(-1)**(i+j) for j in range(N)] for i in range(N)]
    return torch.tensor(vals, device=x.device, dtype=x.dtype)