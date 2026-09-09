def evolved_p5302(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute difference first
    diff = x - y
    
    # Single all_reduce on the difference
    reduced_diff = xm.all_reduce(xm.REDUCE_SUM, diff)
    
    # Check if reduced x > reduced y (i.e., diff > 0)
    return (reduced_diff > 0).to(torch.float32)