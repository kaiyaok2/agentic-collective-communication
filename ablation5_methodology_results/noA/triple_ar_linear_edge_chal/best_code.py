def evolved_p3700(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute linear combination locally first
    local_result = 3 * x + 5 * y + 7 * z
    # Single all_reduce instead of three
    return xm.all_reduce(xm.REDUCE_SUM, local_result)