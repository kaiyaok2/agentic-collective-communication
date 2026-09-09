
def evolved_p164(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = max(i, j)
    # Position-based: each element is the max of its row and column index
    # No collective needed - compute locally
    idx = torch.arange(N, device=x.device)
    i = idx.view(N, 1)  # row indices (N, 1)
    j = idx.view(1, N)  # column indices (1, N)
    result = torch.where(i > j, i, j)
    return result.to(x.dtype)
