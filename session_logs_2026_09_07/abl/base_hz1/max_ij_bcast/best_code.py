
def evolved_p164(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = max(i, j)
    # Position-based: output[row_idx, col_idx] = max(row_idx, col_idx)
    # No collective needed - all ranks compute the same local result
    
    idx = torch.arange(N, device=x.device)
    i = idx.view(N, 1)  # row indices (N, 1)
    j = idx.view(1, N)  # col indices (1, N)
    # max(i, j) = i if i >= j else j
    result = torch.where(i >= j, i, j)  # broadcasts to (N, N)
    return result.to(x.dtype)
