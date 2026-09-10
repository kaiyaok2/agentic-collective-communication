def evolved_p3901(x, y, z, w, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first (cheap local operations)
    local_sum = 1.5 * x + 2.5 * y + 3.5 * z + 4.5 * w
    # Single all_reduce instead of 4 separate ones
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result