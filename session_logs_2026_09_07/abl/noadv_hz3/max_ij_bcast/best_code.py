
def evolved_p164(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = max(i, j)
    # Position-based: no collective needed
    # Create indices [0, 1, 2, ..., N-1]
    idx = torch.arange(N, device=x.device)
    # Broadcast to (N, 1) and (1, N) for row and column indices
    i = idx.unsqueeze(1)  # shape (N, 1)
    j = idx.unsqueeze(0)  # shape (1, N)
    # Apply max(i, j) using torch.where: max(i,j) = i if i >= j else j
    result = torch.where(i >= j, i, j)
    return result.to(x.dtype)
