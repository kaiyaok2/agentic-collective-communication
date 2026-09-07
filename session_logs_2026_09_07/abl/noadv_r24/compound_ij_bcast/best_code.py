
def evolved_p94(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = min(i, j) * max(i, j) + (i - j) ** 2
    # Note: min(i,j) * max(i,j) = i * j always
    # So: x[i, j] = i * j + (i - j)^2
    
    idx = torch.arange(N, device=x.device)
    ii = idx.view(N, 1)  # row indices (N, 1)
    jj = idx.view(1, N)  # column indices (1, N)
    
    diff = ii - jj
    result = ii * jj + diff * diff
    
    return result.to(x.dtype)
