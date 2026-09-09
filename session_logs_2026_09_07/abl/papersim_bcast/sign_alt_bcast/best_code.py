
def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = (-1)^(i + j)
    # Position-based checkerboard: +1 when i+j even, -1 when i+j odd
    # Can be computed locally without any collective communication
    
    idx = torch.arange(N, device=x.device)
    i = idx.view(N, 1)  # row indices (N, 1)
    j = idx.view(1, N)  # col indices (1, N)
    
    # (i + j) % 2 gives 0 for even sum, 1 for odd sum
    # Map: 0 -> +1, 1 -> -1 using formula: 1 - 2 * parity
    parity = (i + j) % 2
    result = 1 - 2 * parity
    
    return result.to(x.dtype)
