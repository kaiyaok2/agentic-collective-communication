def evolved_p4700(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine weighted inputs before reduction
    combined = 2*x1 + 4*x2 + 6*x3 + 8*x4
    # Single all_reduce instead of 4
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    return result