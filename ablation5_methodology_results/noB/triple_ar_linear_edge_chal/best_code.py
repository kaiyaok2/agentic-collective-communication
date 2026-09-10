def evolved_p3700(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine weighted sum locally before reducing
    # This is mathematically equivalent to doing 3 separate reductions
    temp = 3 * x + 5 * y + 7 * z
    result = xm.all_reduce(xm.REDUCE_SUM, temp)
    return result