
def evolved_p94(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = min(i, j) * max(i, j) + (i - j) ** 2
    # Simplified: min(i,j) * max(i,j) = i * j
    # So: x[i, j] = i * j + (i - j) ** 2
    
    idx = torch.arange(N, device=x.device)
    i = idx.view(N, 1)
    j = idx.view(1, N)
    
    diff = i - j
    result = i * j + diff * diff
    return result.to(x.dtype)
