
def evolved_p164(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = max(i, j)
    # Position-based computation - no collective needed
    idx = torch.arange(N, device=x.device)
    ii = idx.view(N, 1)   # row indices (N, 1)
    jj = idx.view(1, N)   # column indices (1, N)
    result = torch.where(ii > jj, ii, jj)  # max(i, j)
    return result.to(x.dtype)
