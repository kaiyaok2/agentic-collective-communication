def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = (-1)^(i + j)
    # Checkerboard: +1 when (i+j) even, -1 when (i+j) odd
    
    # For small N, constant folding might be faster
    if N <= 32:
        # Precompute the checkerboard pattern
        pattern = []
        for i in range(N):
            row = []
            for j in range(N):
                row.append(1 if (i + j) % 2 == 0 else -1)
            pattern.append(row)
        return torch.tensor(pattern, device=x.device, dtype=x.dtype)
    
    # For larger N, use arithmetic
    idx = torch.arange(N, device=x.device)
    i = idx.view(N, 1)
    j = idx.view(1, N)
    result = 1 - 2 * ((i + j) % 2)
    return result.to(x.dtype)