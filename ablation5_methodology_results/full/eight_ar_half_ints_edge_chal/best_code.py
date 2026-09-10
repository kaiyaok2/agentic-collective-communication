def evolved_p4401(x1, x2, x3, x4, x5, x6, x7, x8, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Apply weights locally first, then do a single all_reduce
    # This is mathematically equivalent due to linearity of all_reduce with SUM
    weighted = 0.5 * x1 + 1.5 * x2 + 2.5 * x3 + 3.5 * x4 + 4.5 * x5 + 5.5 * x6 + 6.5 * x7 + 7.5 * x8
    result = xm.all_reduce(xm.REDUCE_SUM, weighted)
    return result