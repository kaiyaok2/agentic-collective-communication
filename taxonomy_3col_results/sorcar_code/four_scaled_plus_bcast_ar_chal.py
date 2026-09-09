def evolved_p6702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine all operations into a single all_reduce
    # Original: 2*sum(x) + 3*sum(x) + 5*sum(x) + 7*sum(x) + sum(ones)
    # = 17*sum(x) + world_size*ones
    sum_x = xm.all_reduce(xm.REDUCE_SUM, x)
    result = 17 * sum_x + world_size
    return result