
def evolved_p164(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = max(i, j)
    # Position-based: depends only on row/col indices, not input values
    # No collective needed - compute locally on each rank
    
    idx = torch.arange(N, device=x.device)
    # Create row indices (N, 1) and col indices (1, N) for broadcasting
    ii = idx.unsqueeze(1)  # shape (N, 1) - row index
    jj = idx.unsqueeze(0)  # shape (1, N) - column index
    
    # Compute max(i, j) using torch.where: max(i,j) = i if i >= j else j
    result = torch.where(ii >= jj, ii, jj)  # shape (N, N)
    
    return result.to(x.dtype)
