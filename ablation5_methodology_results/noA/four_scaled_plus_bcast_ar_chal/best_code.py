def evolved_p6702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 5 separate ones
    sum_x = xm.all_reduce(xm.REDUCE_SUM, x)
    # Compute 17*sum_x + world_size (where world_size comes from reducing ones)
    result = 17 * sum_x + world_size
    return result