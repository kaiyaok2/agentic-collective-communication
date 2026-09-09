def evolved_p116(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_gather, local mean & max
    gathered = xm.all_gather(x, dim=0).reshape(world_size, N)
    mean_val = gathered.mean(dim=0)
    abs_max_val = gathered.abs().max(dim=0).values
    return mean_val / (abs_max_val + 1e-8)
