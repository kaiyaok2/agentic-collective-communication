def evolved_p4502(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute local weighted sum first
    local_sum = 3*x1 - 2*x2 + 5*x3 - 4*x4 + 7*x5 - 6*x6
    
    # Single all_reduce on the combined result
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    
    return result