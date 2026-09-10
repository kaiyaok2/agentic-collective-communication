def evolved_p4502(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute weighted sum locally first, then all_reduce once
    # This is mathematically equivalent due to linearity of all_reduce
    s = 3 * x1 - 2 * x2 + 5 * x3 - 4 * x4 + 7 * x5 - 6 * x6
    result = xm.all_reduce(xm.REDUCE_SUM, s)
    return result