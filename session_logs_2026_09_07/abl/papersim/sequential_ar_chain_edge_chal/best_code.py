
def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try all_gather with stack
    stacked = torch.stack([x, y])  # Shape: (2, N)
    gathered = xm.all_gather(stacked)  # Shape: (world_size * 2 * N,) or similar
    
    # Reshape to (world_size, 2, N)
    gathered = gathered.view(world_size, 2, N)
    
    # Sum across ranks
    summed = gathered.sum(dim=0)  # Shape: (2, N)
    
    # Compute result
    return summed[1] + summed[0] * (2 * world_size)
