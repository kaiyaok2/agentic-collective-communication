def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute linear combination locally first
    local_result = 5 * z + 3 * y + 2 * x
    # Single all_reduce instead of three
    result = xm.all_reduce(xm.REDUCE_SUM, local_result)
    return result