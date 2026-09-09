def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = (-1)^(i + j)
    # Checkerboard: +1 when (i+j) even, -1 when (i+j) odd
    # Position-based → compute locally, no collective needed
    
    idx = torch.arange(N, device=x.device)
    ii = idx.view(N, 1)  # row indices (N, 1)
    jj = idx.view(1, N)  # col indices (1, N)
    
    # (-1)^(i+j) = 1 - 2*((i+j) % 2)
    result = 1 - 2 * ((ii + jj) % 2)
    return result.to(x.dtype)