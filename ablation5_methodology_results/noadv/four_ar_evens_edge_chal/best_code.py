def evolved_p4700(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally
    local_sum = 2 * x1 + 4 * x2 + 6 * x3 + 8 * x4
    # Single all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result