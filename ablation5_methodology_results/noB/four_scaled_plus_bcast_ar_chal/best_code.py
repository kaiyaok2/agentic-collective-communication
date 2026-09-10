def evolved_p6702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    sum_x = xm.all_reduce(xm.REDUCE_SUM, x)
    result = 17 * sum_x + world_size
    return result