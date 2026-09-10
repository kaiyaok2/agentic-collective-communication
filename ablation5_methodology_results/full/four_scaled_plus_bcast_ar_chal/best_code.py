def evolved_p6702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 5
    sum_x = xm.all_reduce(xm.REDUCE_SUM, x)
    # Combine: (2+3+5+7)*sum_x + world_size = 17*sum_x + world_size
    result = 17 * sum_x + world_size
    return result